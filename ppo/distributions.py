import torch
import torch.nn as nn

from ppo.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class FixedMultinomial(torch.distributions.Multinomial):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        batch_size = actions.size(0)
        log_probs = super().log_prob(actions)
        log_probs = log_probs.view(batch_size, -1).sum(-1).unsqueeze(-1)
        return log_probs

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        import warnings
        warnings.warn("DEBUG ME")
        return -p_log_p.sum(-1)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class Multinomial(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        assert all(no == num_outputs[0] for no in num_outputs)
        self.num_outputs = num_outputs
        self.linear = init_(nn.Linear(num_inputs, sum(num_outputs)))

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], len(self.num_outputs), self.num_outputs[0])
        return FixedMultinomial(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
