import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _text_to_token_idx(batch_response, table):
    tokens = []
    for response in batch_response:
        assert len(response) == 1
        for content in response:
            assert content
            content = content.strip().split()
            sentence_idxs = torch.cat([table[word] for word in content])
            assert len(content) == 1, "variable length content not handled, will required EOS token."
            tokens.append(sentence_idxs)
    assert len(tokens) == len(batch_response)
    tokens = torch.stack(tokens).unsqueeze(-1)  # batch x time x 1
    return tokens


class LSTM_DQN(torch.nn.Module):
    batch_size = 1
    seq_input_len = 1

    def __init__(self, model_config, agent_vocab, env_vocab, device, output_length):
        super(LSTM_DQN, self).__init__()
        self.device = device
        self.model_config = model_config
        # self.enable_cuda = enable_cuda
        self.word_vocab_size = len(agent_vocab)
        # self.id2word = word_vocab
        self.output_length = output_length
        self.read_config()
        self._def_layers()
        self.init_weights()

        self.agent_vocab = sorted(agent_vocab)
        self.env_vocab = sorted(env_vocab)

        self.agent_reverse_vocab = {word: torch.tensor([idx], device=self.device) for idx, word in enumerate(self.agent_vocab)}
        self.env_reverse_vocab = {word: torch.tensor([idx], device=self.device) for idx, word in enumerate(self.env_vocab)}
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        self.embedding_size = self.model_config['embedding_size']
        self.encoder_rnn_hidden_size, = self.model_config['encoder_rnn_hidden_size']
        self.action_scorer_hidden_dim = self.model_config['action_scorer_hidden_dim']

    def _def_layers(self):
        self.word_embedding = torch.nn.Embedding(self.word_vocab_size, self.embedding_size, device=self.device)
        self.encoder = torch.nn.GRU(self.embedding_size, self.encoder_rnn_hidden_size, batch_first=True)

        self.Q_features = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_rnn_hidden_size, self.action_scorer_hidden_dim),
            torch.nn.ReLU()
        )
        action_scorers = []
        for _ in range(self.output_length):
            action_scorers.append(torch.nn.Linear(self.action_scorer_hidden_dim, self.word_vocab_size, bias=False))
        self.output_qvalues = torch.nn.ModuleList(action_scorers)
        self.fake_recurrent_mask = None

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Q_features[0].weight.data)
        for i in range(len(self.output_qvalues)):
            torch.nn.init.xavier_uniform_(self.output_qvalues[i].weight.data)
        self.Q_features[0].bias.data.fill_(0)

    def representation_generator(self, response_tokens):
        assert response_tokens.shape in ((self.batch_size, self.seq_input_len, 1), (32, self.seq_input_len, 1))
        # batch x time x emb
        embeddings = self.word_embedding(response_tokens.squeeze(2))  # batch x time x emb
        if response_tokens.shape[1] == 1:
            mask = torch.zeros_like(embeddings)  # for the moment therre is no end of string.
        else:
            raise NotImplementedError("Variable length inptus are not supported yet")

        encoding_sequence, last_state = self.encoder(embeddings)  # , mask)  # batch x time x h
        last_state = last_state.squeeze(0)  # remove the direction * num_layers dim (always 1)

        # mean_encoding = masked_mean(encoding_sequence, mask)  # batch x h
        return last_state

    def get_Q(self, state_representation):
        hidden = self.Q_features(state_representation)  # batch x hid
        actions_q = []
        for output_qvalue in self.output_qvalues:
            actions_q.append(output_qvalue(hidden))  # batch x n_vocab
        actions_q = torch.stack(actions_q, dim=1)  # Batch x output_length x n_vocab
        return actions_q

    def agent_encode(self, batch_response):
        return _text_to_token_idx(batch_response, self.agent_reverse_vocab)

    def env_encode(self, batch_response):
        return _text_to_token_idx(batch_response, self.env_reverse_vocab)
