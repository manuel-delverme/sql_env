import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class SQLModel(nn.Module):
    def __init__(self, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.rnn = nn.RNN(embedding_dim, 128)
        self.linear2 = nn.Linear(128, len(vocab))

    def forward(self, text):
        word_idxes = self.encode(text)

        embeds = self.embeddings(word_idxes).unsqueeze(1)
        _, last_hidden = self.rnn(embeds)
        out = F.relu(last_hidden)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def imprint(self, html):
        self.reference_html = html
