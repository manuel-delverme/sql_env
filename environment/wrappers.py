import collections

import torch

import config


def _text_to_token_idx(batch_response, table):
    tokens = []
    for response in batch_response:
        assert len(response) == 1
        for content in response:
            assert content
            content = content.strip().split()
            sentence_idxs = torch.cat([table[word] for word in content])
            assert len(content) <= (1 + config.cheat_columns + config.cheat_hidden_parameter), "variable length content not handled, will required EOS token."
            tokens.append(sentence_idxs)
    assert len(tokens) == len(batch_response)
    tokens = torch.stack(tokens)  # batch x seq_len
    return tokens


class WordLevelPreprocessing:
    def __init__(self, env, action_history):
        self.env = env

        obs_space, action_space = env.observation_space, env.action_space

        self.obs_vocab = obs_space.vocab
        self.env_reverse_vocab = {word: torch.tensor([idx], device="cpu") for idx, word in enumerate(self.obs_vocab)}

        self.action_vocab = action_space.vocab
        self.action_reverse_vocab = {word: torch.tensor([idx], device="cpu") for idx, word in enumerate(self.action_vocab)}

        self.action_history = collections.deque(
            [torch.zeros(action_space.shape, dtype=torch.int, device="cpu") for _ in range(action_history)], maxlen=action_history)

    def _agent_encode(self, batch_response):
        return _text_to_token_idx(batch_response, self.action_reverse_vocab)

    def _action_decode(self, actions):
        queries = ["".join(self.action_vocab[idx] for idx in query_idx) for query_idx in actions]
        return queries

    def _to_token(self, batch_response):
        return _text_to_token_idx(batch_response, self.env_reverse_vocab)

    def _obs_decode(self, batch_response):
        raise NotImplemented

    def step(self, action):
        self.action_history.append(action.cpu())

        queries = self._action_decode(action)
        next_obs, rewards, dones, infos = self.env.step(queries)
        next_obs_token = self._to_token(next_obs)
        del next_obs
        next_hist_token = torch.cat((next_obs_token, *self.action_history), dim=1)
        return next_hist_token, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        for idx in range(self.action_history.maxlen):
            self.action_history[idx].fill_(0)
        obs_token = self._to_token(obs)
        hist_token = torch.cat((obs_token, *self.action_history), dim=1)
        return hist_token
