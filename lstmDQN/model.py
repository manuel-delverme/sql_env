import torch

import config


class LSTMDQN(torch.nn.Module):
    batch_size = 1

    def __init__(self, obs_vocab_size, action_vocab_size, device, input_length: int, output_length: int):
        super(LSTMDQN, self).__init__()
        self.device = device
        self.obs_vocab_size = obs_vocab_size
        self.action_vocab_size = action_vocab_size
        self.output_length = output_length
        self.embedding_size = config.embedding_size
        self.action_embedding_size = config.embedding_size
        self.encoder_rnn_hidden_size = config.encoder_rnn_hidden_size
        self.action_scorer_hidden_dim = config.action_scorer_hidden_dim

        self.seq_input_len = input_length

        self.word_embedding = torch.nn.Embedding(self.obs_vocab_size, self.embedding_size, device=self.device)
        self.encoder = torch.nn.GRU(self.embedding_size, self.encoder_rnn_hidden_size, batch_first=True)
        self.Q_features = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_rnn_hidden_size, self.action_scorer_hidden_dim),
            torch.nn.ReLU()
        )
        self.output_qvalues = torch.nn.Sequential(
            self.Q_features,
            torch.nn.Linear(self.action_scorer_hidden_dim, self.action_vocab_size * self.output_length),
            torch.nn.Unflatten(-1, (self.output_length, self.action_vocab_size), )
        )
        torch.nn.init.xavier_uniform_(self.Q_features[0].weight.data)
        self.Q_features[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.output_qvalues[-2].weight.data)

    def representation_generator(self, response_tokens):
        assert response_tokens.shape in ((self.batch_size, self.seq_input_len, 1), (32, self.seq_input_len, 1))
        # batch x time x emb
        embeddings = self.word_embedding(response_tokens.squeeze(2))  # batch x time x emb
        encoding_sequence, last_state = self.encoder(embeddings)  # , mask)  # batch x time x h
        last_state = last_state.squeeze(0)  # remove the direction * num_layers dim (always 1)
        return last_state
