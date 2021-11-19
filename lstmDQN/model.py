import torch


class LSTMDQN(torch.nn.Module):
    batch_size = 1

    def __init__(
            self, obs_vocab_size, action_vocab_size, device, output_length: int,
            embedding_size, encoder_rnn_hidden_size, action_scorer_hidden_dim, ):
        super(LSTMDQN, self).__init__()
        self.device = device

        self.word_embedding = torch.nn.Embedding(obs_vocab_size, embedding_size, device=self.device)
        self.encoder = torch.nn.GRU(embedding_size, encoder_rnn_hidden_size, batch_first=True)
        self.Q_features = torch.nn.Sequential(
            torch.nn.Linear(encoder_rnn_hidden_size, action_scorer_hidden_dim),
            torch.nn.ReLU()
        )
        self.output_qvalues = torch.nn.Sequential(
            self.Q_features,
            torch.nn.Linear(action_scorer_hidden_dim, action_vocab_size * output_length),
            torch.nn.Unflatten(-1, (output_length, action_vocab_size), )
        )
        torch.nn.init.xavier_uniform_(self.Q_features[0].weight.data)
        self.Q_features[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.output_qvalues[-2].weight.data)

    def representation_generator(self, response_tokens):
        # batch x time x emb
        embeddings = self.word_embedding(response_tokens.squeeze(2))  # batch x time x emb
        encoding_sequence, last_state = self.encoder(embeddings)  # , mask)  # batch x time x h
        last_state = last_state.squeeze(0)  # remove the direction * num_layers dim (always 1)
        return last_state
