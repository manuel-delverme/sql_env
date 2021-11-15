import torch


def masked_mean(x, m=None, dim=-1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    mask_sum = torch.sum(m, dim=-1)  # batch
    res = torch.sum(x, dim=1)  # batch x h
    res = res / (mask_sum.unsqueeze(-1) + 1e-6)
    return res


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x seq (x is post-padded by 0s)
    outputs:embedding:  batch x seq x emb
            mask:       batch x seq
    '''

    def __init__(self, embedding_size, vocab_size, device):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.device = device
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float().to(device=self.device)
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


# class FastUniLSTM(torch.nn.Module):
#     """
#     Adapted from https://github.com/facebookresearch/DrQA/
#     now supports:   different rnn size for each layer
#                     all zero rows in batch (from time distributed layer, by reshaping certain dimension)
#     """
#
#     def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
#         super(FastUniLSTM, self).__init__()
#         # self.ninp = ninp
#         # self.nhids = nhids
#         # self.nlayers = len(self.nhids)
#
#
#     def forward(self, x, mask):
#         _output, last_state = self.rnns(rnn_input)
#         return last_state
