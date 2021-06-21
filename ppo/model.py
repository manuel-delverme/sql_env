import numpy as np
import torch
import torch.nn as nn

import config


class Policy(nn.Module):
    if config.complexity == 3:
        COST_STR = "a FROM p --"
        voc = [
            " UNION SELECT ",
        ]
    elif config.complexity == 4:
        COST_STR = "FROM p --"
        voc = [
            " UNION SELECT ",
            " NULL, ",
            " a ",
        ]
    elif config.complexity == 5:
        COST_STR = "p --"
        voc = [
            " UNION SELECT ",
            " NULL, ",
            " a ",
            " FROM ",
        ]
    elif config.complexity == 6:
        COST_STR = "--"
        voc = [
            " UNION SELECT ",
            " NULL, ",
            " a ",
            " FROM ",
            " p ",
        ]
    elif config.complexity == 7:
        COST_STR = ""
        voc = [
            " UNION SELECT ",
            " NULL, ",
            " a ",
            " FROM ",
            " p ",
            " -- ",
        ]
    else:
        raise NotImplementedError(f"Complexity {config.complexity} is not implemented.")

    output_vocab = sorted(set(voc).union({
        COST_STR,
        " 1 ",  # escape for int
        " ' ",  # escape for '
        " \" ",  # escape for "
    }))

    def __init__(self, obs_shape, response_vocab, sequence_length, eps):
        super(Policy, self).__init__()
        EMBEDDING_DIM = 10

        # test
        # minial number of token
        self.response_vocab = sorted(response_vocab)

        self.query_word_to_idx = {word: idx for idx, word in enumerate(self.response_vocab)}
        self.output_token_to_idx = {word: idx for idx, word in enumerate(self.output_vocab)}

        self.embeddings_in = nn.Embedding(len(self.response_vocab), EMBEDDING_DIM)
        self.embeddings_in.weight.requires_grad = True

        self.base = MLPBase(EMBEDDING_DIM, len(self.output_vocab), sequence_length, eps=eps)

    def _decode(self, action):
        return " ".join([self.response_vocab[w] for w in action])

    def _encode(self, state):
        retr = torch.tensor([self.output_token_to_idx[w] for w in state], dtype=torch.long)
        return retr

    def act(self, batch_response):
        embeds = self.html_to_embedd(batch_response)
        value, batch_query, query_logprobs, _ = self.base(embeds)
        queries = []
        for query_idx in batch_query:
            query_tokens = [self.output_vocab[torch.argmax(idx)] for idx in query_idx]
            queries.append(query_tokens)
        return value, np.array(queries), query_logprobs

    def html_to_embedd(self, batch_response):
        word_embeddings = []
        for response in batch_response:
            assert len(response) == 1
            for content in response:
                assert content
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
        value, _, _, _ = self.base(embeds)
        return value

    def evaluate_actions(self, batch_response, actions):
        embeds = self.html_to_embedd(batch_response)
        value, query, query_logprobs, concentration = self.base(embeds)
        parsed_actions = []
        for token_sequence in actions:
            for token in token_sequence:
                action_idx = self.output_token_to_idx[token]
                action_vector = torch.zeros(size=(query_logprobs.shape[-1],))
                action_vector[action_idx] = 1
                parsed_actions.append(action_vector)
        parsed_actions = torch.stack(parsed_actions, dim=0).reshape(query_logprobs.shape)

        return value, query_logprobs, parsed_actions, concentration


class MLPBase(nn.Module):
    def __init__(self, num_inputs, dictionary_size, query_length, eps, hidden_size=64):
        super().__init__()
        self._query_length = query_length
        self._hidden_size = hidden_size
        self.gru = nn.GRU(num_inputs, hidden_size)
        self.eps = eps

        self.end_of_line = dictionary_size
        self.actor = AutoregressiveActor(hidden_size, hidden_size, dictionary_size, query_length)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, query_length),
            nn.Softmax(),
        )
        self.train()

    def forward(self, batched_x):
        batch_forwards = []
        for x_i in batched_x:
            assert x_i.ndim == 2
            _, rnn_hxs = self.gru(x_i.unsqueeze(1), None)
            batch_forwards.append(rnn_hxs.squeeze(0))
        rnn_hxs = torch.cat(batch_forwards)
        # TODO: this was _x instead of rnn_hxs, but i want to remove the time dimension
        # TODO fix this with logprop

        value = self.critic(rnn_hxs)
        query_logprobs = self.actor(rnn_hxs)
        concentration = self.prior(rnn_hxs)
        query = torch.distributions.Multinomial(logits=query_logprobs).sample()
        assert query.shape[:2] == query_logprobs.shape[:2]
        return value, query, query_logprobs, concentration


class AutoregressiveActor(nn.Module):
    def __init__(self, num_inputs, hidden_size, dictionary_size, sequence_length):
        super().__init__()
        self.dictionary_size, self.sequence_length = dictionary_size, sequence_length

        self.hidden_to_hidden = nn.Linear(num_inputs, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, dictionary_size * sequence_length)

    def forward(self, hidden):
        output = self.hidden_to_output(hidden)
        output = output.reshape(-1, self.sequence_length, self.dictionary_size)
        output_prob = torch.log_softmax(output, 2)

        return output_prob
